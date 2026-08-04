from __future__ import annotations

import json
from pathlib import Path
import subprocess
import uuid

import pytest
import zarr

from fisheye.analysis_workflows.analysis_candidate_execution import (
    PhysicalIOScope,
    build_candidate_execution_request,
    require_candidate_execution_receipt,
)
from fisheye.analysis_workflows.analysis_candidate_execution_catalog import (
    ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE,
)
from fisheye.analysis_workflows.analysis_candidate_invocation import (
    ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_ID,
    ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_VERSION,
    candidate_invocation_contract_is_frozen,
)
from fisheye.analysis_workflows.tail_posture_candidate_execution import (
    TAIL_POSTURE_EXECUTION_FAMILY_ID,
    TAIL_POSTURE_INVOCATION_CONTRACT_ID,
    build_tail_posture_execution_suite,
    compute_tail_posture_logical_hashes,
)
from fisheye.diagnostics.tail_posture_candidate_execution import (
    TailPostureCandidateExecutionFailed,
    run_tail_posture_candidate_fresh_process,
    tail_posture_selector_snapshot_sha256,
)
from fisheye.shared import tail_coordinate_publication
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_tail_posture_candidate_execution import (
    SOURCE_RUN,
    _invocation_parameters,
    tail_posture_execution_archive,  # noqa: F401
)

_ADAPTER = ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE[
    TAIL_POSTURE_EXECUTION_FAMILY_ID
]
_TAIL_POSTURE_SHARED_INTEGRATED = bool(
    candidate_invocation_contract_is_frozen(TAIL_POSTURE_INVOCATION_CONTRACT_ID)
    and _ADAPTER.runner_status.value == "implemented"
)


def _invocation(parameters: dict[str, object]) -> dict[str, object]:
    payload = {
        "contract_id": TAIL_POSTURE_INVOCATION_CONTRACT_ID,
        "parameters": parameters,
    }
    return {
        "schema_id": ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_ID,
        "schema_version": ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


@pytest.mark.skipif(
    not _TAIL_POSTURE_SHARED_INTEGRATED,
    reason="shared tail-posture invocation/catalog integration is parent-owned",
)
def test_fresh_process_publishes_receipt_and_terminal_failure_attempt(
    tail_posture_execution_archive: Path,  # noqa: F811
) -> None:
    root = zarr.open_group(
        tail_posture_execution_archive,
        mode="r",
        use_consolidated=False,
    )
    source_path = f"analysis/tail_posture_view_runs/{SOURCE_RUN}"
    source = root[source_path]
    publication = tail_coordinate_publication.load_tail_posture_coordinate_publication(
        root,
        source_path,
    )
    source_hashes = compute_tail_posture_logical_hashes(source)
    suite = build_tail_posture_execution_suite(source, repetitions=1)
    invocation = _invocation(_invocation_parameters(publication))
    probe = tail_posture_execution_archive.parent / "protected.json"
    probe.write_text("{}\n", encoding="utf-8")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def request(
        candidate_name: str, *, source_identity_sha256: str
    ) -> dict[str, object]:
        return build_candidate_execution_request(
            execution_id=f"tail_posture_{uuid.uuid4().hex}",
            adapter_manifest=_ADAPTER.as_manifest(),
            invocation=invocation,
            benchmark_suite=suite,
            archive_path=tail_posture_execution_archive,
            source_run_path=source_path,
            candidate_run_path=(f"analysis/tail_posture_view_runs/{candidate_name}"),
            scratch_root=Path("/tmp")
            / f"palette-tail-posture-runner-{uuid.uuid4().hex}",
            source_identity_sha256=source_identity_sha256,
            palette_commit=commit,
            repetition_index=0,
            candidate_order_index=0,
            candidate_order_count=1,
            cache_state="fresh",
            physical_io_scope=PhysicalIOScope.UNAVAILABLE,
            selector_before_sha256=tail_posture_selector_snapshot_sha256(
                tail_posture_execution_archive
            ),
            registry_probe_path=probe,
            production_profiles_probe_path=probe,
        )

    benchmark = tail_posture_execution_archive.parent
    success_request = request(
        "tail_posture_fresh_candidate",
        source_identity_sha256=canonical_json_sha256(source_hashes),
    )
    success_request_path = benchmark / "fresh-request.json"
    success_request_path.write_text(
        json.dumps(success_request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path = benchmark / "fresh-receipt.json"
    attempt_path = benchmark / "fresh-attempt.json"
    receipt = run_tail_posture_candidate_fresh_process(
        success_request_path,
        receipt_path=receipt_path,
        attempt_path=attempt_path,
    )
    require_candidate_execution_receipt(
        receipt,
        expected_request_payload_digest=success_request["payload_digest"],
    )
    assert receipt_path.is_file()
    assert not attempt_path.exists()
    assert receipt["payload"]["logical_equality"] == {
        "contract_id": "tail_posture_v3_arrays_v1",
        "compared_array_count": 10,
        "source_logical_manifest_sha256": canonical_json_sha256(source_hashes),
        "candidate_logical_manifest_sha256": canonical_json_sha256(source_hashes),
        "equal": True,
    }
    assert [phase["phase"] for phase in receipt["payload"]["phases"]] == [
        "plan",
        "source_staging",
        "scientific_compute",
        "local_validation",
        "local_consolidation",
        "local_direct_consolidated_comparison",
        "atomic_publication",
        "published_validation",
        "published_direct_consolidated_comparison",
        "decoded_equality",
        "physical_inventory",
    ]
    final = zarr.open_group(
        tail_posture_execution_archive,
        mode="r",
        use_consolidated=False,
    )
    candidate = final["analysis/tail_posture_view_runs/tail_posture_fresh_candidate"]
    assert candidate.attrs["palette_run_completion_status"] == "complete"
    assert candidate.attrs["stage_selector_eligible"] is False
    assert candidate.attrs["storage_candidate_profile_promoted"] is False

    failed_request = request(
        "tail_posture_failed_candidate",
        source_identity_sha256="0" * 64,
    )
    failed_request_path = benchmark / "failed-request.json"
    failed_request_path.write_text(
        json.dumps(failed_request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed_receipt_path = benchmark / "failed-receipt.json"
    failed_attempt_path = benchmark / "failed-attempt.json"
    with pytest.raises(
        TailPostureCandidateExecutionFailed,
        match="immutable attempt record",
    ):
        run_tail_posture_candidate_fresh_process(
            failed_request_path,
            receipt_path=failed_receipt_path,
            attempt_path=failed_attempt_path,
        )
    assert not failed_receipt_path.exists()
    assert failed_attempt_path.is_file()
    attempt = json.loads(failed_attempt_path.read_text(encoding="utf-8"))
    assert attempt["payload"]["status"] == "failed"
    assert attempt["payload"]["failure_phase"] == "runner_preflight"
    assert attempt["payload"]["nonmutation_evidence"]["unchanged"] is True
