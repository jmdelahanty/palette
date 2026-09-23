"""Admission/refusal contract, captured before the maintained implementation."""

from __future__ import annotations

import json

import h5py
import pytest

from fisheye.shared.unified_h5 import (
    UnifiedH5ContractError,
    validate_unified_h5_artifact,
)
from tests.unit.fisheye.unified_h5_fixtures import (
    emit_fixture,
    receipt_for,
    synthetic_receipt_for_mutated_test_file,
)


@pytest.mark.parametrize(
    "name,count,appearance", [("base", 79, 0), ("appearance", 89, 2)]
)
def test_real_producer_success_without_companion(tmp_path, name, count, appearance):
    path = emit_fixture(tmp_path, name)
    with h5py.File(path, "r") as source:
        evidence = validate_unified_h5_artifact(
            source, source_h5=path, finalization_receipt=receipt_for(name)
        )
        assert evidence.profile == "unified_experimental_h5_v1"
        assert evidence.dependency_count == count
        assert evidence.frame_count == 4
        assert evidence.component_rows["chaser"] == 4
        assert evidence.component_rows.get("visual_appearance", 0) == appearance
        assert (
            evidence.geometry_scope
            == "static_internal_geometry_consistency_not_physical_registration_or_experiment_admission"
        )
        assert not evidence.selector_eligible


@pytest.mark.parametrize(
    "name",
    [
        "full_bound_pair_correspondence_failed",
        "full_bound_pair_deferred_mapping_failed",
        "full_bound_pair_deferred_app_shutdown",
        "full_bound_pair_interrupted_finalization",
    ],
)
def test_diagnostic_files_never_inherit_success(tmp_path, name):
    path = emit_fixture(tmp_path, name)
    with h5py.File(path, "r") as source:
        with pytest.raises(UnifiedH5ContractError):
            validate_unified_h5_artifact(
                source, source_h5=path, finalization_receipt=receipt_for("base")
            )
        assert source["/frames/stimulus"].shape == (4,)


def test_wrong_external_receipt_refused(tmp_path):
    path = emit_fixture(tmp_path)
    with h5py.File(path, "r") as source, pytest.raises(UnifiedH5ContractError):
        validate_unified_h5_artifact(
            source, source_h5=path, finalization_receipt=receipt_for("base")
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_profile",
        "pending",
        "invalid_boolean",
        "missing_appearance",
        "external_link",
        "cycle",
        "protocol_tamper",
    ],
)
def test_invalid_artifact_refused_even_with_test_only_byte_reseal(tmp_path, mutation):
    path = emit_fixture(tmp_path)
    with h5py.File(path, "r+") as source:
        if mutation == "unknown_profile":
            source["/metadata/session"].attrs[
                "recording_artifact_profile"
            ] = "future_v99"
        elif mutation == "pending":
            value = json.loads(source["/metadata/completion_json"][()])
            value["finalization"] = "pending"
            source["/metadata/completion_json"][()] = json.dumps(value).encode()
        elif mutation == "invalid_boolean":
            data = source["/components/visual_appearance/states"][()]
            data["drive_valid"][0] = 2
            source["/components/visual_appearance/states"][:] = data
        elif mutation == "missing_appearance":
            del source["/components/visual_appearance/states"]
        elif mutation == "external_link":
            source["external"] = h5py.ExternalLink("must-not-open.h5", "/")
        elif mutation == "cycle":
            source["cycle"] = source["/"]
        else:
            source["/protocol/authored/protocol_definition_json"][()] = b"{}"
    receipt = synthetic_receipt_for_mutated_test_file(path)
    with h5py.File(path, "r") as source, pytest.raises(UnifiedH5ContractError):
        validate_unified_h5_artifact(
            source, source_h5=path, finalization_receipt=receipt
        )


def test_wrong_open_handle_refused(tmp_path):
    first = emit_fixture(tmp_path, "appearance")
    second = emit_fixture(tmp_path, "base")
    with h5py.File(first, "r") as source, pytest.raises(UnifiedH5ContractError):
        validate_unified_h5_artifact(
            source, source_h5=second, finalization_receipt=receipt_for()
        )
