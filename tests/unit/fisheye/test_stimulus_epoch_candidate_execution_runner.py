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
    build_stimulus_epoch_invocation,
)
from fisheye.analysis_workflows.stimulus_epoch_candidate_execution import (
    build_stimulus_epoch_execution_suite,
    compute_stimulus_epoch_logical_hashes,
    compute_stimulus_group_logical_fingerprint,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.diagnostics.stimulus_epoch_candidate_execution import (
    StimulusEpochCandidateExecutionFailed,
    require_stimulus_epoch_execution_attempt,
    run_stimulus_epoch_candidate_fresh_process,
    stimulus_epoch_selector_snapshot_sha256,
)
from tests.unit.fisheye.test_stimulus_epoch_schema import (
    create_legacy_stimulus_epoch_archive,
)

SOURCE_PATH = "analysis/stimulus_epoch_runs/source"
CANDIDATE_PATH = "analysis/stimulus_epoch_runs/typed_candidate_v2"


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
    fingerprint: str | None = None,
) -> dict[str, object]:
    root = zarr.open_group(
        str(archive),
        mode="r",
        zarr_format=3,
        use_consolidated=False,
    )
    source = root[SOURCE_PATH]
    stimulus = root[str(source.attrs["source_stimulus_path"])]
    live_fingerprint = compute_stimulus_group_logical_fingerprint(stimulus)
    adapter = ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE["stimulus_epochs"]
    return build_candidate_execution_request(
        execution_id=f"stimulus_epoch_{uuid.uuid4().hex}",
        adapter_manifest=adapter.as_manifest(),
        invocation=build_stimulus_epoch_invocation(
            source_stimulus_fingerprint=fingerprint or live_fingerprint,
            source_epoch_lineage_hash=str(source.attrs["lineage_hash"]),
            storage_profile_id="published_http_v1",
            copy_backend="python",
            keep_scratch=False,
        ),
        benchmark_suite=build_stimulus_epoch_execution_suite(
            source,
            repetitions=1,
        ),
        archive_path=archive,
        source_run_path=SOURCE_PATH,
        candidate_run_path=CANDIDATE_PATH,
        scratch_root=scratch,
        source_identity_sha256=canonical_json_sha256(
            compute_stimulus_epoch_logical_hashes(source)
        ),
        palette_commit=_git_commit(),
        repetition_index=0,
        candidate_order_index=0,
        candidate_order_count=1,
        cache_state="fresh_process_os_cache_uncontrolled",
        physical_io_scope=PhysicalIOScope.UNAVAILABLE,
        selector_before_sha256=stimulus_epoch_selector_snapshot_sha256(archive),
        registry_probe_path=probe,
        production_profiles_probe_path=probe,
    )


@pytest.mark.parametrize("valid_fingerprint", [True, False])
def test_fresh_process_executes_or_fails_closed(
    tmp_path: Path,
    valid_fingerprint: bool,
) -> None:
    benchmark_root = tmp_path / ".palette_benchmarks" / "stimulus-epoch"
    benchmark_root.mkdir(parents=True)
    archive = benchmark_root / "fixture.zarr"
    create_legacy_stimulus_epoch_archive(archive)
    probe = benchmark_root / "protected-probe.json"
    probe.write_text("{}\n", encoding="utf-8")
    scratch = Path("/tmp") / f"palette-stimulus-test-{uuid.uuid4().hex}"
    request = _request(
        archive,
        scratch=scratch,
        probe=probe,
        fingerprint=None if valid_fingerprint else "f" * 64,
    )
    workflow = benchmark_root / f"driver-{valid_fingerprint}"
    evidence_dir = workflow / "evidence"
    evidence_dir.mkdir(parents=True)
    request_path = workflow / "request.json"
    receipt_path = evidence_dir / "receipt.json"
    attempt_path = evidence_dir / "attempt.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    if valid_fingerprint:
        evidence = run_stimulus_epoch_candidate_fresh_process(
            request_path,
            receipt_path=receipt_path,
            attempt_path=attempt_path,
        )
        require_candidate_execution_receipt(
            evidence,
            expected_request_payload_digest=request["payload_digest"],
        )
        assert evidence["payload"]["coordinate_evidence"]["status"] == (
            "verified_temporal_axis"
        )
        assert (
            evidence["payload"]["coordinate_evidence"]["coordinate_gate_passed"] is True
        )
        assert evidence["payload"]["logical_equality"]["compared_array_count"] == 12
        assert evidence["payload"]["publication_gate_passed"] is False
        assert receipt_path.is_file()
        assert not attempt_path.exists()
        candidate = zarr.open_group(
            str(archive),
            mode="r",
            zarr_format=3,
            use_consolidated=False,
        )[CANDIDATE_PATH]
        assert candidate.attrs["stage_selector_eligible"] is False
        assert candidate.attrs["storage_candidate_profile_promoted"] is False
        assert candidate.attrs["source_staging_mode"] == (
            "epoch_and_stimulus_logical_copy_v1"
        )
    else:
        with pytest.raises(
            StimulusEpochCandidateExecutionFailed,
            match="stimulus group fingerprint",
        ) as raised:
            run_stimulus_epoch_candidate_fresh_process(
                request_path,
                receipt_path=receipt_path,
                attempt_path=attempt_path,
            )
        evidence = raised.value.attempt
        require_stimulus_epoch_execution_attempt(
            evidence,
            expected_request_payload_digest=request["payload_digest"],
        )
        assert evidence["payload"]["failure_phase"] == "runner_preflight"
        assert evidence["payload"]["target_state"]["exists"] is False
        assert attempt_path.is_file()
        assert not receipt_path.exists()
    assert not scratch.exists()


def test_catalog_resolves_exact_stimulus_epoch_runner() -> None:
    adapter = ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE["stimulus_epochs"]

    assert adapter.runner_status.value == "implemented"
    assert adapter.resolves_candidate_owner() is True
    assert adapter.resolves_runner() is True
    assert adapter.resolves_suite_validator() is True
