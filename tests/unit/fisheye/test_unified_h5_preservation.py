"""Pinned producer bytes are independent positive oracles for the new reader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.unit.fisheye.unified_h5_fixtures import FIXTURES, emit_fixture, fixture_bytes


@pytest.mark.parametrize(
    "name", sorted(json.loads((FIXTURES / "inventory.json").read_text())["fixtures"])
)
def test_vendored_producer_artifact_hashes(name):
    assert fixture_bytes(name)


def test_appearance_preserves_all_34_members_and_counts(tmp_path):
    old = emit_fixture(tmp_path, "legacy_appearance")
    new = emit_fixture(tmp_path, "appearance")
    with h5py.File(old, "r") as legacy, h5py.File(new, "r") as unified:
        before = legacy["/tracking_data/object_appearance_states"][()]
        after = unified["/components/visual_appearance/states"][()]
        assert len(before) == len(after) == 2
        assert before.dtype.names == after.dtype.names
        assert len(after.dtype.names) == 34
        for name in before.dtype.names:
            np.testing.assert_array_equal(before[name], after[name])
        for name in (
            "expected_object_appearance_states_rows",
            "tracking_object_appearance_states_rows",
        ):
            assert legacy.attrs[name] == unified["/metadata/session"].attrs[name] == 2
        for name in legacy["/enums"]:
            if name.startswith("appearance_"):
                np.testing.assert_array_equal(
                    legacy[f"/enums/{name}"][()],
                    unified[f"/definitions/enums/{name}"][()],
                )


def test_optional_appearance_absence_is_real_old_fixture(tmp_path):
    with h5py.File(emit_fixture(tmp_path, "base"), "r") as source:
        assert "/components/visual_appearance/states" not in source
        assert source["/frames/stimulus"].shape == (4,)
        assert source["/components/chaser/states"].shape == (4,)


def test_golden_fixture_hash_is_contract_pin():
    assert hashlib.sha256(
        (FIXTURES / "object_appearance_golden_vectors_v1.json").read_bytes()
    ).hexdigest() == (
        "ea1a9be5c3ed2b7d750a14bcd70b1eed6e26f9e9b9f6df88153c11d3be5774f4"
    )


def test_unchanged_reference_evaluator_matches_all_pinned_golden_vectors():
    from fisheye.shared.unified_h5.vendor import object_appearance_reference as oracle

    assert hashlib.sha256(Path(oracle.__file__).read_bytes()).hexdigest() == (
        "e77ead403cee979db26d7b0a453e432f0c5f08f1c2e1aa3ea99b494e5c503f30"
    )
    vectors = json.loads(
        (FIXTURES / "object_appearance_golden_vectors_v1.json").read_bytes()
    )
    for case in vectors["vectors"]:
        result = oracle.evaluate_request(case["request"])["result"]
        for name, expected in case["expected"].items():
            if type(expected) is float or (
                type(expected) is list and name != "realized_rgba8"
            ):
                np.testing.assert_allclose(
                    result[name], expected, atol=vectors["absolute_tolerance"], rtol=0
                )
            else:
                assert result[name] == expected, (case["name"], name)
