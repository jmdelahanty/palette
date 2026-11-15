"""Unified H5 admission revision 4: frame identity v1, table-version-map selection.

agent-contracts PR 88 (merged 9e37316; witnesses at 32a3a70) and Citrus PR 24
(merged 53b68a7; catalog 526c218, policy pin 43f57c1). A capacity preflight v2
file is selected by its full {core table path -> schema_version} map, its
declared catalog_id must name the same catalog, and its declared producer
policy must be the one bound to it. Preflight v1 files keep revision 3's pair
key. Witnesses are Citrus's frozen bytes, pinned by digest; the pose-on
witness came from agent-contracts PR 89 (head 60856f3, open at pin time).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5 import pose
from fisheye.shared.unified_h5.correspondence import (
    BINDING,
    PREFLIGHT,
    RECEIPT,
    validate_capacity_preflight,
    validate_component_correspondence,
)
from fisheye.shared.unified_h5.display_submissions import (
    DISPLAY_SUBMISSIONS_PATH,
    DISPLAY_TIMING_PATH,
    build_stimulus_display_submissions,
)
from fisheye.shared.unified_h5.geometry import validate_geometry
from fisheye.shared.unified_h5.integrity import validate_internal_integrity
from fisheye.shared.unified_h5.protocol import validate_protocol
from fisheye.shared.unified_h5.rows import validate_table_relations
from fisheye.shared.unified_h5.schema import (
    ADMISSIONS_BY_PREFLIGHT_VERSION,
    CONTRACT_HASHES,
    CORE_CATALOG,
    CORE_CATALOG_FILES,
    CORRESPONDENCE_RECEIPT_SCHEMAS,
    admitted_binding,
    catalog_table_versions,
    contract,
    read_json,
    table_schema,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "citrus_unified_h5_frame_identity_v1"
COMPLETE = "frame_identity_v1_complete.h5"
POSE = "frame_identity_v1_pose_complete.h5"
PARTIAL = "frame_identity_v1_partial_adoption_refused.h5"
DIGESTS = {
    COMPLETE: "1a21dcbfce84bf0315a5a4919c27185ae906bbd8b27c21fdefb6447579ddfcad",
    PARTIAL: "d346004688cb6cba855a550735148affe7c4d6d41ae67a7211500acf21f3429c",
    POSE: "f2728d607409a172ce64d9cb4927e87fd66e2a47b46b00d4c1becdaae9392b8f",
}
FRAME_IDENTITY_MAP = {
    "/frames/stimulus": 2,
    "/components/chaser/states": 3,
    "/components/visual_appearance/states": 1,
    "/components/independent_motion_grid/states": 1,
    "/components/moving_grating/states": 2,
    "/correspondence/frames/sources": 1,
    "/correspondence/{component}/sources": 1,
    "/events/records": 2,
    "/trials/trial_index": 2,
    "/definitions/enums/{component}": 1,
    "/observations/bounding_boxes": 3,
    "/observations/region_routing": 2,
    "/observations/region_candidates": 2,
    "/timing/display_submissions/frame_submissions": 2,
}
V4 = ADMISSIONS_BY_PREFLIGHT_VERSION[2]


def _canonical_sha256(value) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _copy(tmp_path: Path, name: str = COMPLETE) -> Path:
    return Path(shutil.copy(FIXTURES / name, tmp_path / name))


def _rewrite_preflight(path: Path, **changes) -> None:
    with h5py.File(path, "r+") as h5:
        document = json.loads(h5[PREFLIGHT][()])
        for key, value in changes.items():
            if value is None:
                document.pop(key, None)
            else:
                document[key] = value
        del h5[PREFLIGHT]
        h5.create_dataset(
            PREFLIGHT,
            data=json.dumps(document, sort_keys=True, separators=(",", ":")),
            dtype=h5py.string_dtype("utf-8"),
        )


def _binding(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        document = json.loads(h5[PREFLIGHT][()])
        return admitted_binding(h5, document["admission_policy_sha256"], document)


# --- pins -------------------------------------------------------------------


def test_vendored_witnesses_are_the_published_bytes():
    for name, digest in DIGESTS.items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest


def test_packaged_contracts_match_the_revision_4_pins():
    admission = contract(V4)
    assert admission["schema_version"] == 4
    pins = admission["core_definition_pins"]
    assert pins["selection"]["key"] == "table_version_map"
    pin = pins["frame_identity_v1"]
    assert "sha256:" + CONTRACT_HASHES["experimental_h5_core_frame_identity_v1.json"] == pin["catalog_file_sha256"]
    catalog = contract("experimental_h5_core_frame_identity_v1.json")
    assert _canonical_sha256(catalog) == pin["catalog_canonical_json_sha256"]
    assert pin["table_version_map"] == FRAME_IDENTITY_MAP
    assert catalog_table_versions("experimental_h5_core_frame_identity_v1.json") == FRAME_IDENTITY_MAP
    for path, expected in pin["table_definitions"].items():
        definition = table_schema(path, expected["schema_version"])
        definition["path"] = path
        assert definition["id"] == expected["schema_id"]
        assert _canonical_sha256(definition) == expected["definition_canonical_json_sha256"], path
    binding = admission["admitted_catalog_bindings"]["frame_identity_v1"]
    assert binding["catalog_id"] == "frame_identity_v1"
    assert "sha256:" + CONTRACT_HASHES["unified_h5_producer_policy_3c9866ed.json"] == binding["producer_policy"]["file_sha256"]
    assert (
        "sha256:" + CONTRACT_HASHES["experimental_h5_capacity_preflight_v2.schema.json"]
        == binding["capacity_preflight_schema"]["file_sha256"]
    )
    receipt = admission["correspondence_definition_pins"]["receipt_schema"]
    assert receipt["schema_version"] == 3
    assert "sha256:" + CONTRACT_HASHES[CORRESPONDENCE_RECEIPT_SCHEMAS[3]] == receipt["file_sha256"]
    # Revision 3's pair-keyed pins and bindings carry over unchanged.
    v3 = contract(ADMISSIONS_BY_PREFLIGHT_VERSION[1])
    for name in ("chaser_v2", "bounding_boxes_v2"):
        assert admission["admitted_catalog_bindings"][name] == v3["admitted_catalog_bindings"][name]
        old, new = dict(v3["core_definition_pins"][name]), dict(pins[name])
        assert new.pop("catalog_id") is None
        assert new == old


def test_a_version_names_one_definition_across_every_pinned_catalog():
    # Shared table-independent sections (component outcomes and attributes).
    reference = {k: v for k, v in contract(CORE_CATALOG).items() if k != "tables"}
    for name in CORE_CATALOG_FILES:
        assert {k: v for k, v in contract(name).items() if k != "tables"} == reference, name
        for table in contract(name)["tables"]:
            definition = table_schema(table["path"], table["schema_version"])
            definition["path"] = table["path"]
            assert definition == table, (name, table["path"])


# --- the witnesses ------------------------------------------------------------


@pytest.mark.parametrize("name", [COMPLETE, POSE])
def test_a_frame_identity_v1_witness_is_admitted(name):
    with h5py.File(FIXTURES / name, "r") as h5:
        integrity = validate_internal_integrity(h5)
        validate_capacity_preflight(h5)
        validate_table_relations(h5, integrity.table_descriptors)
        correspondence = validate_component_correspondence(h5)
        validate_protocol(h5)
        validate_geometry(h5, read_json(h5, BINDING, canonical=True), integrity.table_descriptors)
        assert json.loads(h5[RECEIPT][()])["schema_version"] == 3
        assert correspondence.mapped_frame_count == h5["/frames/stimulus"].shape[0]
        for path, descriptor in integrity.table_descriptors.items():
            template = (
                "/definitions/enums/{component}" if path.startswith("/definitions/enums/")
                else "/correspondence/{component}/sources" if path == "/correspondence/chaser/sources"
                else path
            )
            assert descriptor["table"]["schema_version"] == FRAME_IDENTITY_MAP[template], path
    binding = _binding(FIXTURES / name)
    assert binding["catalog"] == "frame_identity_v1"
    assert binding["admission"] == V4
    assert binding["preflight_schema"] == "experimental_h5_capacity_preflight_v2.schema.json"
    assert binding["correspondence_receipt_version"] == 3


def test_the_complete_witness_has_no_pose_component():
    with h5py.File(FIXTURES / COMPLETE, "r") as h5:
        assert validate_internal_integrity(h5).pose_rows is None


def test_the_pose_witness_admits_pose_v2_through_the_heading_checks(monkeypatch):
    calls = []
    original = pose._check_v2_heading

    def spy(objects):
        calls.append(objects)
        return original(objects)

    monkeypatch.setattr(pose, "_check_v2_heading", spy)
    with h5py.File(FIXTURES / POSE, "r") as h5:
        assert json.loads(h5[pose.CONTRACT][()])["schema_version"] == 2
        assert json.loads(h5[pose.RECEIPT][()])["schema_version"] == 2
        assert validate_internal_integrity(h5).pose_rows == 7
    assert len(calls) == 1


def test_a_pose_witness_row_that_breaks_the_heading_convention_is_refused(tmp_path):
    path = _copy(tmp_path, POSE)
    with h5py.File(path, "r+") as h5:
        objects = h5["/observations/pose/objects"]
        record = objects[0]
        record["heading_angle_degrees"] = 90.0  # direction (1, 0) implies 0 degrees
        objects[0] = record
    with h5py.File(path, "r") as h5:
        data = json.loads(h5["/metadata/component_outcomes_json"][()])
        outcomes = {row["component_id"]: row for row in data["components"]}
        with pytest.raises(UnifiedH5ContractError, match="pose_heading_angle_disagrees_with_direction"):
            pose.validate_pose_observations(h5, outcomes, {pose.COMPONENT})


def test_the_partial_adoption_witness_is_refused_without_falling_back():
    with h5py.File(FIXTURES / PARTIAL, "r") as h5:
        with pytest.raises(UnifiedH5ContractError, match="catalog_id_mismatch"):
            validate_capacity_preflight(h5)
        # Its frames table no longer matches its own canonical descriptor either.
        with pytest.raises(UnifiedH5ContractError):
            validate_internal_integrity(h5)
        document = json.loads(h5[PREFLIGHT][()])
        # Not even under the bounding_boxes_v2 policy and a v1 preflight.
        for policy in ("b2af7764c7f27f6295007ba3049dcf8a35d0f84c5801f8a1ff56150c7f9e7fc1",
                       document["admission_policy_sha256"]):
            with pytest.raises(UnifiedH5ContractError):
                admitted_binding(h5, policy, {**document, "schema_version": 1})


# --- map-keyed selection on mutated copies -------------------------------------


@pytest.mark.parametrize(
    ("path", "version"),
    [
        ("/components/chaser/states", 2),  # the revision 3 pair, otherwise frame identity
        ("/observations/bounding_boxes", 2),
        ("/events/records", 1),
        ("/timing/display_submissions/frame_submissions", 1),
    ],
)
def test_partial_adoption_is_a_catalog_id_mismatch(tmp_path, path, version):
    copy = _copy(tmp_path)
    with h5py.File(copy, "r+") as h5:
        h5[path].attrs.modify("schema_version", np.uint64(version))
    with h5py.File(copy, "r") as h5, pytest.raises(UnifiedH5ContractError, match="catalog_id_mismatch"):
        validate_capacity_preflight(h5)


def test_a_version_in_no_admitted_map_is_refused(tmp_path):
    copy = _copy(tmp_path)
    with h5py.File(copy, "r+") as h5:
        h5["/frames/stimulus"].attrs.modify("schema_version", np.uint64(7))
    with h5py.File(copy, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="unsupported_table_version:/frames/stimulus"
    ):
        validate_capacity_preflight(h5)


@pytest.mark.parametrize("catalog_id", ["bounding_boxes_v2", "frame_identity_v2", None])
def test_a_catalog_id_that_names_another_or_no_binding_is_refused(tmp_path, catalog_id):
    copy = _copy(tmp_path)
    _rewrite_preflight(copy, catalog_id=catalog_id)
    with h5py.File(copy, "r") as h5, pytest.raises(UnifiedH5ContractError, match="catalog_id_mismatch"):
        validate_capacity_preflight(h5)


def test_another_bindings_policy_is_refused(tmp_path):
    copy = _copy(tmp_path)
    _rewrite_preflight(
        copy, admission_policy_sha256="b2af7764c7f27f6295007ba3049dcf8a35d0f84c5801f8a1ff56150c7f9e7fc1"
    )
    with h5py.File(copy, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="capacity_preflight_producer_policy"
    ):
        validate_capacity_preflight(h5)


def test_an_unknown_preflight_version_is_refused(tmp_path):
    copy = _copy(tmp_path)
    # Preflight v3 selects revision 5 (angle_v1), v4 revision 6 (phototaxis_v1)
    # and v5 revision 7 (video_overlay_v1); 6 is unknown.
    _rewrite_preflight(copy, schema_version=6)
    with h5py.File(copy, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="capacity_preflight_unknown_schema_version"
    ):
        validate_capacity_preflight(h5)


def test_a_preflight_v1_frame_identity_file_is_refused(tmp_path):
    """Revision 3 has no pair for (3, 3): a v1 preflight cannot admit the new catalog."""
    copy = _copy(tmp_path)
    _rewrite_preflight(copy, schema_version=1, catalog_id=None)
    with h5py.File(copy, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="capacity_preflight_producer_policy"
    ):
        validate_capacity_preflight(h5)


def test_an_undeclared_absent_table_does_not_constrain(tmp_path):
    copy = _copy(tmp_path)
    with h5py.File(copy, "r+") as h5:
        del h5["/observations/region_candidates"]
        document = json.loads(h5[PREFLIGHT][()])
    _rewrite_preflight(
        copy,
        tables=[t for t in document["tables"] if t["path"] != "/observations/region_candidates"],
    )
    assert _binding(copy)["catalog"] == "frame_identity_v1"


def test_a_present_undeclared_table_is_refused(tmp_path):
    """Undeclared tables do not select, but a present core table must be declared."""
    copy = _copy(tmp_path)
    with h5py.File(copy, "r+") as h5:
        h5["/observations/region_candidates"].attrs.modify("schema_version", np.uint64(1))
        document = json.loads(h5[PREFLIGHT][()])
    _rewrite_preflight(
        copy,
        tables=[t for t in document["tables"] if t["path"] != "/observations/region_candidates"],
    )
    with h5py.File(copy, "r") as h5, pytest.raises(UnifiedH5ContractError, match="declared_core_table_mismatch"):
        validate_capacity_preflight(h5)


def test_a_revision_3_file_with_a_frame_identity_table_is_refused(tmp_path):
    """The pair key ignores frames; the selected catalog still pins every table's version."""
    source = Path(__file__).resolve().parents[2] / "fixtures" / "citrus_unified_h5_bounding_boxes_v2"
    copy = Path(shutil.copy(source / "bounding_boxes_v2_capacity_pose_off.h5", tmp_path / "mixed.h5"))
    with h5py.File(copy, "r+") as h5:
        h5["/frames/stimulus"].attrs.modify("schema_version", np.uint64(2))
    with h5py.File(copy, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="table_version_not_in_selected_catalog:/frames/stimulus"
    ):
        validate_capacity_preflight(h5)


# --- correspondence receipt v3 ---------------------------------------------------


def _rewrite_receipt(path: Path, **changes) -> None:
    with h5py.File(path, "r+") as h5:
        receipt = json.loads(h5[RECEIPT][()])
        receipt.update(changes)
        data = h5[RECEIPT]
        dtype = data.dtype
        del h5[RECEIPT]
        h5.create_dataset(
            RECEIPT, data=json.dumps(receipt, sort_keys=True, separators=(",", ":")), dtype=dtype
        )


def test_a_receipt_v2_under_revision_4_is_refused(tmp_path):
    copy = _copy(tmp_path)
    _rewrite_receipt(copy, schema_version=2)
    with h5py.File(copy, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="correspondence_receipt_revision_mismatch"
    ):
        validate_component_correspondence(h5)


def test_an_unknown_receipt_version_is_refused(tmp_path):
    copy = _copy(tmp_path)
    _rewrite_receipt(copy, schema_version=4)
    with h5py.File(copy, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="correspondence_receipt_unknown_schema_version"
    ):
        validate_component_correspondence(h5)


# --- display submissions v2 ---------------------------------------------------------


def test_the_witness_display_submissions_v2_reduce_per_state():
    with h5py.File(FIXTURES / COMPLETE, "r") as h5:
        table = h5[DISPLAY_SUBMISSIONS_PATH]
        assert "source_camera_frame_id" not in table.dtype.names
        attrs = {
            name: (value.decode() if isinstance(value, bytes) else value.item() if hasattr(value, "item") else value)
            for name, value in table.attrs.items()
        }
        parent = {
            name: (value.item() if hasattr(value, "item") else value)
            for name, value in h5[DISPLAY_TIMING_PATH].attrs.items()
        }
        result = build_stimulus_display_submissions(
            table[()],
            h5["/frames/stimulus"]["stimulus_frame_num"],
            table_attrs=attrs,
            parent_attrs=parent,
            source_reference_sha256="witness",
        )
    assert attrs["schema_version"] == 2
    assert result.submitted.all()
    assert result.first_display_frame_index.tolist() == [100, 101, 102, 103]
    assert result.first_swap_end_ns_session.tolist() == [1030, 1130, 1230, 1330]
