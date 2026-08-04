from __future__ import annotations

import json
from pathlib import Path
import subprocess
import uuid

import pytest
import zarr

from fisheye.analysis.chaser_distance_base_storage import (
    build_source_authority_binding,
)
from fisheye.analysis.chaser_distance_coordinate_publication import (
    load_bound_chaser_distance_run,
)
from fisheye.analysis_workflows.analysis_candidate_execution import (
    PhysicalIOScope,
    build_candidate_execution_request,
    require_candidate_execution_receipt,
)
from fisheye.analysis_workflows.analysis_candidate_execution_catalog import (
    ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE,
)
from fisheye.analysis_workflows.analysis_candidate_invocation import (
    build_chaser_distance_base_invocation,
)
from fisheye.analysis_workflows.chaser_distance_candidate_execution import (
    build_chaser_distance_execution_suite,
    chaser_distance_decoded_identity_sha256,
)
from fisheye.diagnostics.chaser_distance_candidate_execution import (
    ChaserDistanceCandidateExecutionFailed,
    chaser_distance_selector_snapshot_sha256,
    require_chaser_distance_execution_attempt,
    run_chaser_distance_candidate_fresh_process,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from tests.unit.fisheye.test_chaser_distance_coordinate_publication import (
    _publish_canonical,
)

SOURCE_PATH = "analysis/chaser_distance_runs/canonical_distance"
CANDIDATE_PATH = "analysis/chaser_distance_storage_candidates/typed_candidate_v2"


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _request(
    archive: Path,
    *,
    scratch: Path,
    probe: Path,
    source_binding_sha256: str | None = None,
) -> dict[str, object]:
    root = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=False
    )
    source = root[SOURCE_PATH]
    bound = load_bound_chaser_distance_run(root, SOURCE_PATH)
    binding = build_source_authority_binding(bound, source_group=source)
    adapter = ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE["chaser_distance"]
    return build_candidate_execution_request(
        execution_id=f"chaser_distance_{uuid.uuid4().hex}",
        adapter_manifest=adapter.as_manifest(),
        invocation=build_chaser_distance_base_invocation(
            source_authority_binding_sha256=(
                source_binding_sha256 or canonical_json_sha256(binding)
            ),
            storage_profile_id="published_http_v1",
            copy_backend="python",
            keep_scratch=False,
        ),
        benchmark_suite=build_chaser_distance_execution_suite(
            source,
            repetitions=1,
        ),
        archive_path=archive,
        source_run_path=SOURCE_PATH,
        candidate_run_path=CANDIDATE_PATH,
        scratch_root=scratch,
        source_identity_sha256=chaser_distance_decoded_identity_sha256(source),
        palette_commit=_git_commit(),
        repetition_index=0,
        candidate_order_index=0,
        candidate_order_count=1,
        cache_state="fresh_process_os_cache_uncontrolled",
        physical_io_scope=PhysicalIOScope.UNAVAILABLE,
        selector_before_sha256=chaser_distance_selector_snapshot_sha256(archive),
        registry_probe_path=probe,
        production_profiles_probe_path=probe,
    )


@pytest.mark.parametrize("valid_binding", [True, False])
def test_fresh_process_executes_or_fails_closed(
    tmp_path: Path,
    valid_binding: bool,
) -> None:
    benchmark_root = tmp_path / ".palette_benchmarks" / "chaser-distance"
    benchmark_root.mkdir(parents=True)
    archive, _root, _run = _publish_canonical(benchmark_root)
    consolidate_metadata_capture_expected_warnings(archive)
    probe = benchmark_root / "protected-probe.json"
    probe.write_text("{}\n", encoding="utf-8")
    scratch = Path("/tmp") / f"palette-chaser-test-{uuid.uuid4().hex}"
    request = _request(
        archive,
        scratch=scratch,
        probe=probe,
        source_binding_sha256=None if valid_binding else "f" * 64,
    )
    workflow = benchmark_root / f"driver-{valid_binding}"
    evidence_dir = workflow / "evidence"
    evidence_dir.mkdir(parents=True)
    request_path = workflow / "request.json"
    receipt_path = evidence_dir / "receipt.json"
    attempt_path = evidence_dir / "attempt.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    if valid_binding:
        evidence = run_chaser_distance_candidate_fresh_process(
            request_path,
            receipt_path=receipt_path,
            attempt_path=attempt_path,
        )
        require_candidate_execution_receipt(
            evidence,
            expected_request_payload_digest=request["payload_digest"],
        )
        coordinate = evidence["payload"]["coordinate_evidence"]
        assert coordinate["status"] == ("verified_source_preservation_nonminting")
        assert coordinate["coordinate_gate_passed"] is False
        assert evidence["payload"]["logical_equality"]["compared_array_count"] == 30
        assert evidence["payload"]["publication_gate_passed"] is False
        assert receipt_path.is_file()
        assert not attempt_path.exists()
        candidate = zarr.open_group(
            str(archive), mode="r", zarr_format=3, use_consolidated=False
        )[CANDIDATE_PATH]
        assert candidate.attrs["stage_selector_eligible"] is False
        assert candidate.attrs["storage_candidate_profile_promoted"] is False
        assert candidate.attrs["source_staging_mode"] == ("sealed_base_logical_copy_v1")
    else:
        with pytest.raises(
            ChaserDistanceCandidateExecutionFailed,
            match="source authority binding",
        ) as raised:
            run_chaser_distance_candidate_fresh_process(
                request_path,
                receipt_path=receipt_path,
                attempt_path=attempt_path,
            )
        evidence = raised.value.attempt
        require_chaser_distance_execution_attempt(
            evidence,
            expected_request_payload_digest=request["payload_digest"],
        )
        assert evidence["payload"]["failure_phase"] == "runner_preflight"
        assert evidence["payload"]["target_state"]["exists"] is False
        assert attempt_path.is_file()
        assert not receipt_path.exists()
    assert not scratch.exists()


def test_catalog_resolves_chaser_runner_and_distinct_source_parent() -> None:
    adapter = ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE["chaser_distance"]
    manifest = adapter.as_manifest()["payload"]

    assert adapter.runner_status.value == "implemented"
    assert manifest["source_run_parent"] == "analysis/chaser_distance_runs"
    assert manifest["run_parent"] == ("analysis/chaser_distance_storage_candidates")
    assert adapter.resolves_candidate_owner() is True
    assert adapter.resolves_runner() is True
    assert adapter.resolves_suite_validator() is True
