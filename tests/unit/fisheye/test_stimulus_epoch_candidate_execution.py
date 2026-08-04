from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fisheye.analysis_workflows.stimulus_epoch_candidate_execution import (
    STIMULUS_EPOCH_EXECUTION_ARRAY_COUNT,
    build_stimulus_epoch_source_identity,
    build_stimulus_epoch_temporal_axis_evidence,
    build_stimulus_epoch_execution_suite,
    compute_stimulus_group_logical_fingerprint,
    require_stimulus_epoch_execution_suite,
    stimulus_epoch_source_identity_sha256,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_stimulus_epoch_schema import (
    create_legacy_stimulus_epoch_archive,
)


def _suite(tmp_path: Path) -> dict[str, object]:
    root = create_legacy_stimulus_epoch_archive(tmp_path / "fixture.zarr")
    return build_stimulus_epoch_execution_suite(
        root["analysis/stimulus_epoch_runs/source"],
        seed=23,
        repetitions=2,
    )


def test_suite_reconstructs_exact_twelve_array_v2_projection(tmp_path: Path) -> None:
    suite = _suite(tmp_path)

    require_stimulus_epoch_execution_suite("stimulus_epochs", suite)
    payload = suite["payload"]
    assert payload["repetitions"] == 2
    assert len(payload["storage_plan_receipt"]["payload"]["arrays"]) == (
        STIMULUS_EPOCH_EXECUTION_ARRAY_COUNT
    )


def test_suite_rejects_another_family_and_rehashed_declaration(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path)
    with pytest.raises(ValueError, match="owns only"):
        require_stimulus_epoch_execution_suite("eye_angles", suite)

    changed = deepcopy(suite)
    receipt = changed["payload"]["storage_plan_receipt"]
    record = receipt["payload"]["arrays"][0]
    record["declaration"]["fill_semantics"] = "tampered"
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    changed["payload_digest"] = canonical_json_sha256(changed["payload"])
    with pytest.raises(ValueError, match="declaration|digest|storage plan"):
        require_stimulus_epoch_execution_suite("stimulus_epochs", changed)


def test_source_identity_and_temporal_axis_bind_stimulus_and_epoch(
    tmp_path: Path,
) -> None:
    root = create_legacy_stimulus_epoch_archive(tmp_path / "identity.zarr")
    source_path = "analysis/stimulus_epoch_runs/source"
    source = root[source_path]
    stimulus = root[str(source.attrs["source_stimulus_path"])]
    fingerprint = compute_stimulus_group_logical_fingerprint(stimulus)

    identity = build_stimulus_epoch_source_identity(
        source,
        source_stimulus_fingerprint=fingerprint,
    )
    evidence = build_stimulus_epoch_temporal_axis_evidence(
        source_run_path=source_path,
        source_group=source,
        source_stimulus_fingerprint=fingerprint,
    )

    assert stimulus_epoch_source_identity_sha256(
        source,
        source_stimulus_fingerprint=fingerprint,
    ) == canonical_json_sha256(identity)
    assert len(identity["source_array_logical_hashes"]) == 12
    assert evidence["status"] == "verified_temporal_axis"
    assert evidence["coordinate_gate_passed"] is True
    assert evidence["temporal_axis_ref"] == (
        f"/{source_path}#stimulus_epoch_temporal_axis_v1"
    )
    assert [item["role"] for item in evidence["source_authority_digests"]] == [
        "source_stimulus_epoch_lineage",
        "source_stimulus_logical_tree",
    ]
