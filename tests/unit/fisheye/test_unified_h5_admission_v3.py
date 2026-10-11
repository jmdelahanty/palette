"""Unified H5 admission revision 3: bounding_boxes v2 and per-catalog policy bindings.

agent-contracts PR 73 (fcce821): the file's versioned tables and its declared
producer policy select one admitted catalog binding; its own preflight schema
then applies. Fixtures are Citrus's frozen bytes, pinned by digest.
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
from fisheye.shared.unified_h5.correspondence import PREFLIGHT, validate_capacity_preflight
from fisheye.shared.unified_h5.integrity import validate_internal_integrity
from fisheye.shared.unified_h5.schema import (
    ADMISSION,
    BOUNDING_BOXES,
    CONTRACT_HASHES,
    admitted_binding,
    contract,
    table_schema,
)
from tests.unit.fisheye.unified_h5_fixtures import emit_fixture

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "citrus_unified_h5_bounding_boxes_v2"
DIGESTS = {
    "bounding_boxes_v2_capacity_pose_off.h5": "13a67cbffc3a54c9d5e661ca58dd80dd77a543aa24ab852605b8f7fc21aea663",
    "bounding_boxes_v2_capacity_pose_on.h5": "f7b1d6a663e630e4397544540b3e521e7bcd4b1f31975d57d49284f30a43bb0a",
    "bounding_boxes_v2_pose_off.h5": "6920631211ea0f38bdcd0edd0f2c7edd3b82a6ec3491e06e545048628ca32722",
    "bounding_boxes_v2_pose_on.h5": "116ebfd97790c0cfdff0c161e300ae44dbfc946cf7a61f4b6dc679b6dea4db48",
}


def _canonical_sha256(value) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def test_vendored_fixtures_are_the_published_bytes():
    for name, digest in DIGESTS.items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest


def test_packaged_contracts_match_the_revision_3_pins():
    admission = contract(ADMISSION)
    pins = admission["core_definition_pins"]
    for name, pin in (
        ("experimental_h5_core_chaser_v2.json", pins["chaser_v2"]),
        ("experimental_h5_core_bounding_boxes_v2.json", pins["bounding_boxes_v2"]),
        ("experimental_h5_core_v1.json", pins["historical_v1"]),
    ):
        assert "sha256:" + CONTRACT_HASHES[name] == pin["catalog_file_sha256"], name
    bindings = admission["admitted_catalog_bindings"]
    for catalog, policy, schema in (
        ("chaser_v2", "unified_h5_producer_policy_d544b081.json",
         "experimental_h5_capacity_preflight_v1.schema.json"),
        ("bounding_boxes_v2", "unified_h5_producer_policy_b2af7764.json",
         "experimental_h5_capacity_preflight_v1_ba8333a3.schema.json"),
    ):
        assert "sha256:" + CONTRACT_HASHES[policy] == bindings[catalog]["producer_policy"]["file_sha256"]
        assert (
            "sha256:" + CONTRACT_HASHES[schema]
            == bindings[catalog]["capacity_preflight_schema"]["file_sha256"]
        )
    for version, catalog in ((1, "chaser_v2"), (2, "bounding_boxes_v2")):
        pin = pins[catalog]
        assert pin["table_versions"]["bounding_boxes"] == version
        definition = table_schema(BOUNDING_BOXES, version)
        assert definition["schema_version"] == version


@pytest.mark.parametrize("name", ["bounding_boxes_v2_capacity_pose_off.h5", "bounding_boxes_v2_pose_off.h5"])
def test_a_bounding_boxes_v2_file_is_admitted(name):
    with h5py.File(FIXTURES / name, "r") as h5:
        validate_internal_integrity(h5)
        if PREFLIGHT in h5:
            validate_capacity_preflight(h5)
            declared = json.loads(h5[PREFLIGHT][()])["admission_policy_sha256"]
            assert admitted_binding(h5, declared)["catalog"] == "bounding_boxes_v2"


def test_a_chaser_v2_file_keeps_its_revision_2_policy():
    path = emit_fixture(Path(__import__("tempfile").mkdtemp()), "admission_v2_roundtrip")
    with h5py.File(path, "r") as h5:
        validate_capacity_preflight(h5)
        declared = json.loads(h5[PREFLIGHT][()])["admission_policy_sha256"]
        assert admitted_binding(h5, declared)["catalog"] == "chaser_v2"


def test_the_new_policy_with_bounding_boxes_v1_is_refused(tmp_path):
    path = Path(shutil.copy(FIXTURES / "bounding_boxes_v2_capacity_pose_off.h5", tmp_path / "mixed.h5"))
    with h5py.File(path, "r+") as h5:
        h5[BOUNDING_BOXES].attrs.modify("schema_version", np.uint64(1))
    with h5py.File(path, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="capacity_preflight_producer_policy"
    ):
        validate_capacity_preflight(h5)


def test_the_old_policy_with_bounding_boxes_v2_is_refused(tmp_path):
    path = emit_fixture(tmp_path, "admission_v2_roundtrip")
    with h5py.File(path, "r+") as h5:
        h5[BOUNDING_BOXES].attrs.modify("schema_version", np.uint64(2))
    with h5py.File(path, "r") as h5, pytest.raises(
        UnifiedH5ContractError, match="capacity_preflight_producer_policy"
    ):
        validate_capacity_preflight(h5)
