from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fisheye.analysis.chaser_distance_base_storage import (
    build_source_authority_binding,
)
from fisheye.analysis_workflows.chaser_distance_candidate_execution import (
    CHASER_DISTANCE_EXECUTION_ARRAY_COUNT,
    build_chaser_distance_execution_suite,
    build_chaser_distance_source_preservation_evidence,
    chaser_distance_decoded_identity_sha256,
    require_chaser_distance_execution_suite,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_chaser_distance_base_candidate import (
    _archive,
    _bound,
)


def test_suite_reconstructs_exact_sealed_30_array_projection(tmp_path: Path) -> None:
    root = _archive(tmp_path / "suite.zarr")
    source = root["analysis/chaser_distance_runs/source"]
    suite = build_chaser_distance_execution_suite(source, repetitions=2)

    require_chaser_distance_execution_suite("chaser_distance", suite)
    assert suite["payload"]["repetitions"] == 2
    assert len(suite["payload"]["storage_plan_receipt"]["payload"]["arrays"]) == (
        CHASER_DISTANCE_EXECUTION_ARRAY_COUNT
    )
    assert len(chaser_distance_decoded_identity_sha256(source)) == 64


def test_suite_rejects_rehashed_declaration_tampering(tmp_path: Path) -> None:
    root = _archive(tmp_path / "tamper.zarr")
    suite = build_chaser_distance_execution_suite(
        root["analysis/chaser_distance_runs/source"],
        repetitions=1,
    )
    changed = deepcopy(suite)
    receipt = changed["payload"]["storage_plan_receipt"]
    receipt["payload"]["arrays"][0]["declaration"]["fill_semantics"] = "tampered"
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    changed["payload_digest"] = canonical_json_sha256(changed["payload"])

    with pytest.raises(ValueError, match="declaration|storage plan|digest"):
        require_chaser_distance_execution_suite("chaser_distance", changed)


def test_source_preservation_evidence_binds_all_seven_authorities(
    tmp_path: Path,
) -> None:
    root = _archive(tmp_path / "authority.zarr")
    source = root["analysis/chaser_distance_runs/source"]
    binding = build_source_authority_binding(_bound(), source_group=source)
    evidence = build_chaser_distance_source_preservation_evidence(binding)

    assert evidence["status"] == "verified_source_preservation_nonminting"
    assert evidence["coordinate_gate_passed"] is False
    assert [item["role"] for item in evidence["source_authority_digests"]] == [
        "chaser_collection",
        "epoch_window_identity",
        "input_authority",
        "measurement_authority",
        "publication_seal",
        "row_identity",
        "surface_manifest",
    ]
